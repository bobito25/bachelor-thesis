

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(ontable c)
(ontable d)
(on e c)
(on f a)
(clear b)
(clear d)
(clear f)
)
(:goal
(and
(on b c)
(on c f)
(on e b)
(on f d))
)
)


