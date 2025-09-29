

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(ontable c)
(ontable d)
(ontable e)
(on f a)
(clear b)
(clear c)
(clear d)
(clear f)
)
(:goal
(and
(on c e)
(on d c))
)
)


