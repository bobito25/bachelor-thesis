

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a c)
(ontable b)
(ontable c)
(on d a)
(ontable e)
(on f d)
(clear b)
(clear e)
(clear f)
)
(:goal
(and
(on a b)
(on c d)
(on d a)
(on e f))
)
)


